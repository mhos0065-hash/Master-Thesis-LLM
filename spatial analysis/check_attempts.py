import json

path = r"runs\20260903T000757Z_spatialcorr_multiagent_meta-llama-llama-3.1-8b-instruct_moran_local_2021_waste\attempts_log.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

attempts = data.get("attempts", data if isinstance(data, list) else [])
print("Total attempts:", len(attempts))
print()
for i, a in enumerate(attempts):
    cr = a.get("coder_result", {})
    print("--- Attempt", i + 1, "(plan=", a.get("plan_number"), ", try=", a.get("coder_try"), ") ---")
    print("validation_score:", a.get("validation_score"), "/", a.get("validation_total"))
    print("failed_checks:", a.get("failed_checks"))
    print("final_answer:", repr(cr.get("final_answer")))
    print()