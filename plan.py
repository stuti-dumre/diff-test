"""Turn a plan file into the job lists of the workflow."""
import json
import sys

plan = json.load(open(sys.argv[1]))


def shards(jobs):
    return [dict(j, shard=f"{i}/{j['parts']}") for j in jobs for i in j.get("only", range(j["parts"]))]


train = [{"tier": plan["tier"], "steps": plan["steps"], **t, "minutes": plan["max_minutes"]} for t in plan["train"]]
print("train=" + json.dumps(train))
print("gen_old=" + json.dumps([dict(g, steps=plan["gen_steps"]) for g in shards(plan["gen_old"])]))
print("gen_new=" + json.dumps([dict(g, steps=plan["gen_steps"]) for g in shards(plan["gen_new"])]))
