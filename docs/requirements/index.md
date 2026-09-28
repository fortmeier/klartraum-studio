# Requirements

Requirements, specifications and tests are tracked with
[sphinx-needs](https://sphinx-needs.readthedocs.io/), in the same way as in
the [Klartraum Engine documentation](https://fortmeier.github.io/klartraum/docs/requirements/index.html).

| Type | Prefix | Links |
|---|---|---|
| Requirement | `REQ_` | |
| Specification | `SPEC_` | `implements` a requirement |
| Test | `TEST_` | `verifies` a specification or requirement; `gtest` names the GoogleTest case |

Statuses: `draft`, `open`, `implemented`, `verified`.

```{note}
This section is at an early stage. The items below are a starting set that
shows the structure; they are not a complete specification of the studio.
```

## Overview

```{needtable}
:columns: id, title, type, status, implements, verifies
:style: table
```

## Compiled graph view

```{req} Show what Klartraum compiled
:id: REQ_VIEW_COMPILED
:status: implemented

The compiled graph view shall show exactly the elements Klartraum compiled
for the live graph or the last run.
```

```{test} Introspected graph matches the compiled elements
:id: TEST_VIEW_COMPILED_MATCHES
:status: verified
:verifies: REQ_VIEW_COMPILED
:gtest: GraphIntrospection.builtGraphMatchesCompiledElements

Checks for both splatting backends that the introspected graph contains
exactly the elements Klartraum compiled.
```
