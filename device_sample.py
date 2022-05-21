import argparse
import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

from datetime import datetime
from datetime import date


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    params = json.load(open(args.config))
    
    

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    


    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        while True:
            decision = input("1: for manual-gen\n2: for auto-gen\nYour choice: ")
            if decision == "1":
                while True:
                    context = input("btpm404 or Type input:")
                    if context == "btpm404":
                        break
                    tokens = tokenizer.encode(context)

                    start = time.time()

                    provided_ctx = len(tokens)
                    pad_amount = seq - provided_ctx
                    pad_amount = max(pad_amount, 0)
                    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)[-2048:]
                    batched_tokens = np.array([padded_tokens] * total_batch)
                    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

                    temperature = input("Type temperature:")
                    try:
                        temperature = float(temperature)
                    except(ValueError):
                        temperature = 0.75

                    top_p = input("Type top_p:")
                    try:
                        top_p = float(top_p)
                    except(ValueError):
                        top_p = 0.9

                    out_length = input("Type output length:")
                    try:
                        out_length = int(out_length)
                    except(ValueError):
                        out_length = 512

                    output = network.generate(batched_tokens, length, out_length, {"top_p": np.ones(total_batch) * top_p,
                                                                            "temp": np.ones(total_batch) * temperature})

                    for idx, o in enumerate(output[1][0][:, :, 0]):
                        string = repr(tokenizer.decode(o))
                        string = string.replace(r"\n", "\n")
                        print(f"sample {idx}: {string}\n")

                    print(f"completion done in {time.time() - start:06}s")
            elif decision == "2":
                try:
                    sample = open("data/prompts.json", "r")
                    list = json.load(sample)
                    sample.close()
                except(FileNotFoundError):
                    print("prompts not found")
                    continue

                temperature = input("Type temperature:")
                try:
                    temperature = float(temperature)
                except(ValueError):
                    temperature = 0.75

                top_p = input("Type top_p:")
                try:
                    top_p = float(top_p)
                except(ValueError):
                    top_p = 0.9

                out_length = input("Type output length:")
                try:
                    out_length = int(out_length)
                except(ValueError):
                    out_length = 512

                inference_out ={
                    "starting datetime": str(date.today()) + " " +(datetime.now()).strftime("%H:%M:%S"),
                    "finishing datetime": "",
                    "Temperature":  temperature,
                    "Top_p": top_p,
                    "Out_length": out_length,
                    "tasks_count": len(list),
                    "sample_no": per_replica_batch,
                    "incomplete_generations": 0,
                    "tasks": []
                    }
                tasks = []
                start = time.time()
                sum_time =0
                times_count=0
                try:
                    for i in range(len(list)):   
                        task_id = list[i]['task_id']
                        task_description = list[i]['task_description']
                        prompt = list[i]['prompt']
                        test_list = list[i]['test_list']

                        sample = {
                            "task_id": task_id,
                            "task_description": task_description ,
                            "test_list":test_list,
                            "sample_id": "",
                            "completion": "",	
                        }
                        context = prompt
                        tokens = tokenizer.encode(context)

                        min_start = time.time()

                        provided_ctx = len(tokens)
                        pad_amount = seq - provided_ctx
                        pad_amount = max(pad_amount, 0)
                        padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)[-2048:]
                        batched_tokens = np.array([padded_tokens] * total_batch)
                        length = np.ones(total_batch, dtype=np.uint32) * len(tokens)
                        
                        
                        output = network.generate(batched_tokens, length, out_length, {"top_p": np.ones(total_batch) * top_p,
                                                                                "temp": np.ones(total_batch) * temperature})
                                    #  generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):

                        
                        for idx, o in enumerate(output[1][0][:, :, 0]):
                            sample = {
                            "task_id": task_id,
                            "task_description": task_description ,
                            "test_list":test_list,
                            "sample_id": idx,
                            "completion": repr(tokenizer.decode(o))	
                            }
                            tasks.append(sample)
                            # print(f"sample {idx}: {str}\n")
                        sum_time += time.time() - min_start
                        times_count+=1
                        print(f"completion done in {time.time() - min_start:06}s")
                        print("iteration", i,"/",len(list),"eta: ",(len(list)-i)*(sum_time/times_count)/60," min")
                except Exception as e:
                    print("XXXXXXXXXXXXXXXX--Error occured--XXXXXXXXXXXXXXXX")
                    print(e)
                    pass
                print(f"Generation done in {time.time() - start:06}s")
                print("Saving...")
                inference_out["tasks"] = tasks
                inference_out["finishing datetime"] = str(date.today()) + " " +(datetime.now()).strftime("%H:%M:%S")
                try:
                    inference_stream = open("unfiltered_generation.json",mode= "w", encoding='utf-8')
                    inference_stream.write(json.dumps(inference_out))
                    inference_stream.close()
                except(FileNotFoundError):
                    print("unfiltered_generation.json not found")
                    continue