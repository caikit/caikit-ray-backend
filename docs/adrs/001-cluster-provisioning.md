# ADR 001: Ray Cluster Provisioning

The current implementation of this module makes the assumption that a Ray cluster is pre-existing and is managed by the user. Each training is submitted to this pre-existing Ray cluster as a job. The storage of training job state is delegated entirely to this Ray cluster.

### Current state: Assume existing Ray cluster

Pros:

*  Simple (from the perspective of this module, at least)
* Is platform agnostic. i.e. The Ray cluster could be a user's laptop, KubeRay, AWS offerings, etc.

Cons:

* Shifts complexity to user
* A static Ray cluster could be squatting on precious GPUs when not actively training

Possible mitigating enhancements:

* Include cluster creation and management to an operator such as the Caikit operator.
* Have clusters autoscale, in order to minimize resource wastage. (Need to test how this works in real life)

### Option: Spin up a new Ray cluster for every training

Pros:
* This allows the size and resources of the cluster to be fully customizable
* Could be useful for very large training/tuning jobs

Cons:
* Caikit-ray-backend now cesases to be platform agnostic, as it will have to embed logic for creating Ray clusters in onre or more platform-specific ways (i.e. K8s, AWS, GCP.. etc)
* We will need to add state management to the caikit ray module to keep pointers to the various Ray clusters
* Ray clusters must explicitly be deleted. They will squat on precious resources like GPUs until they are deleted. Once a Ray cluster is deleted, information about the job run is also deleted. If we want to persist it, it must be stored in the Caikit Ray backend's own data store.
* Total overkill for things like single GPU / single node prompt tuning jobs


## Decision

Since our current use cases are for `caikit-nlp` tuning jobs that run within one node, we will assume a **pre-existing** Ray cluster. Operators that install caikit into a K8s environemnt can take responsibility for pre-creating the Ray cluster for use of Caikit tuning jobs.


## Status

Approved


## Consequences

* caikit-ray-backend remains platform agnostic
* If or when caikit runs very large multi-node tuning jobs, we may have to revisit this decision
* The responsibility of creating Ray clusters falls on the user of Caikit (and/or future operators in the K8s context)