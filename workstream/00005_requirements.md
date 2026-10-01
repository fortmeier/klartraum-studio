# 00005 Requirements for tensors, images and graphs

Deferred on purpose ("don't write the requirements just yet").

## Scope, once it's time

- Where tensors and images are interchangeable in the node graph, and where they need a
  conversion node.
- The live graph versus the Run graph: what each may depend on, and how results flow
  from the Run into the live graph (retained results).
- Meta nodes: nesting, built-in meta nodes, and navigating into and out of them.
