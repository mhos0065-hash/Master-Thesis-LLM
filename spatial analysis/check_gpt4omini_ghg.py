import json
import glob
import os

# Find the most recent run folder for this exact chunk + model
pattern = r"runs\*spatialcorr_multiagent_openai-gpt-4o-mini_moran_local_2021_ghg"
matches = glob.glob(pattern)
if not matches:
    print("No matching run folder found for pattern:", pattern)
else:
    latest = max(matches, key=os.path.getmtime)
    print("Using run folder:", latest)
    print()
    path = os.path.join(latest, "attempts_log.json")
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
        print("data_points_used:", cr.get("data_points_used"))
        print("final_answer:", repr(cr.get("final_answer")))
        print()
