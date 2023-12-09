# -*- coding: utf-8 -*-

from transformers import FlaxAutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from flax.training.common_utils import get_metrics, onehot, shard
from flax.training import train_state
import math
import jax.numpy as jnp
import flax
import optax
import jax
from pathlib import Path
from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from pathlib import Path
from transformers import AutoConfig
from transformers import AutoTokenizer
import time

language = "is"
model_config = "distilgpt2"
model_dir = model_config + f"-pretrained-{language}"


Path(model_dir).mkdir(parents=True, exist_ok=True)


config = AutoConfig.from_pretrained(model_config)
config.save_pretrained(f"{model_dir}")

raw_dataset = load_dataset("oscar", f"unshuffled_deduplicated_{language}")

tokenizer = ByteLevelBPETokenizer()


def batch_iterator(batch_size=1000):
    for i in range(0, len(raw_dataset), batch_size):
        yield raw_dataset["train"][i: i + batch_size]["text"]


tokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])


tokenizer.save(f"{model_dir}/tokenizer.json")


max_seq_length = 512


raw_dataset["train"] = load_dataset(
    "oscar", f"unshuffled_deduplicated_{language}", split="train[5%:]")
raw_dataset["validation"] = load_dataset(
    "oscar", f"unshuffled_deduplicated_{language}", split="train[:5%]")

# these cells should be commented out to run on full dataset

raw_dataset["train"] = raw_dataset["train"].select(range(10000))
raw_dataset["validation"] = raw_dataset["validation"].select(range(1000))


tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = raw_dataset.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=raw_dataset["train"].column_names)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i: i + max_seq_length]
            for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_datasets = tokenized_datasets.map(
    group_texts, batched=True, num_proc=4)


per_device_batch_size = 16
num_epochs = 3
training_seed = 0
learning_rate = 3e-4

total_batch_size = per_device_batch_size * jax.device_count()
print("total batch size: {}".format(total_batch_size))
num_train_steps = len(
    tokenized_datasets["train"]) // total_batch_size * num_epochs

init_start = time.time()
model = FlaxAutoModelForCausalLM.from_config(
    config, seed=training_seed, dtype=jnp.dtype("bfloat16"))


# model = FlaxAutoModelForCausalLM.from_config(config, seed=training_seed)

linear_decay_lr_schedule_fn = optax.linear_schedule(
    init_value=learning_rate, end_value=0, transition_steps=num_train_steps)


adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn,
                    b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)


state = train_state.TrainState.create(
    apply_fn=model.__call__, params=model.params, tx=adamw)

# print("init_time: %.4f" % (time.time()-init_start))

def data_loader(rng, dataset, batch_size, shuffle=False):
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    # Skip incomplete batch.
    batch_idx = batch_idx[: steps_per_epoch * batch_size]
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch


def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=params,
                                dropout_rng=dropout_rng, train=True)[0]

        loss = optax.softmax_cross_entropy(
            logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng


parallel_train_step = jax.jit(jax.pmap(train_step, "batch"))


def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    loss = optax.softmax_cross_entropy(
        logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])).mean()

    # summarize metrics
    metrics = {"loss": loss, "perplexity": jnp.exp(loss)}
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics


parallel_eval_step = jax.pmap(eval_step, "batch")


state = flax.jax_utils.replicate(state)

rng = jax.random.PRNGKey(training_seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())


for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    rng, input_rng = jax.random.split(rng)
    epoch_start = time.time()
    # -- Train --
    train_loader = data_loader(input_rng, tokenized_datasets["train"], total_batch_size, shuffle=True)
    with tqdm(total=len(tokenized_datasets["train"]) // total_batch_size, desc="Training...", leave=False) as progress_bar_train:
        for model_inputs in train_loader:
            # Model forward
            state, train_metric, dropout_rngs = parallel_train_step(
                state, model_inputs, dropout_rngs)
            # state, train_metric, dropout_rngs = train_step(
            #     state, model_inputs, dropout_rngs)
            progress_bar_train.update(1)

        progress_bar_train.write(
            f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
        )
    print("epoch: %d, time %.4f" % (epoch, time.time()-epoch_start))
    # -- Eval --
    eval_loader = data_loader(
        input_rng, tokenized_datasets["validation"], total_batch_size)
    eval_metrics = []

    with tqdm(total=len(tokenized_datasets["validation"]) // total_batch_size, desc="Evaluation...", leave=False) as progress_bar_eval:
        for model_inputs in eval_loader:
            # Model forward
            eval_metric = parallel_eval_step(state.params, model_inputs)
            eval_metrics.append(eval_metric)

            progress_bar_eval.update(1)

        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
        progress_bar_eval.write(
            f"Eval... ({epoch}/{num_epochs} | Loss: {eval_metrics['loss']} | Perplexity: {eval_metrics['perplexity']})"
        )
