# 19 - Coupled systems and solver stability

Coupled linear system:
[ x' ] = [A  B][x] + [f]
[ y' ]   [C  D][y]   [g]

Stability depends on:
- spectrum of the block operator
- coupling strength
- time stepping scheme

In code: `tig.diffeq.coupled_systems`, `tig.diffeq.time_stepping`.
