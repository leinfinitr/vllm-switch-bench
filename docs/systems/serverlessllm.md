# ServerlessLLM

**Status:** blocked for v0.1 numeric publication pending a current-source automatic scale-to-zero run.

ServerlessLLM is a serverless loading baseline. The relevant current lifecycle is the configured automatic `keep_alive=0` scale-to-zero path, not merely deleting registry metadata. The legacy adapter's delete/register and model-absence checks are retained for debugging but are not the v0.1 publication boundary.

## Required gate

A publishable run must use a current-source, digest-bound image and prove:

1. register and complete a strict inference;
2. automatic scale-to-zero removes backend actors/processes;
3. scheduler GPU reservation returns;
4. aggregate and process-level GPU residency reaches the calibrated idle threshold;
5. the model remains registered if that is the automatic-stop contract;
6. the next request reloads the model and completes strict inference;
7. cleanup also handles startup failure before an instance reaches the ready set.

## Current blockers in retained evidence

- current-source image builds were blocked by registry/network or dependency-download failures;
- an older image removed registry metadata while a backend kept GPU memory;
- a source overlay exposed failed-start actor/scheduler-reservation cleanup gaps;
- no retained current-source automatic scale-to-zero cycle satisfies all post-conditions.

Therefore ServerlessLLM remains visible as `blocked` and contributes no latency row to the v0.1 release plot. Do not convert delete acknowledgements or old-image results into scale-to-zero numbers.

A local run requires a sanitized external runtime configuration and explicit host/container model paths. Keep all machine paths in ignored local files and retain the final image digest and source commit in raw metadata.
