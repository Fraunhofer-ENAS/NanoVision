Experiments Package
===================

Purpose
-------
This package contains exploratory analyses, visualizations, and one-off
research scripts.

Algorithms that become part of the canonical preprocessing pipeline should be
migrated into `src/cnt_project/preprocessing/` together with validation,
schemas, and runners.

Current status
--------------

✓ Density
    Migrated to preprocessing.metadata.density

✓ Wavelet noise
    Migrated to preprocessing.metadata.noise

✓ Length metadata
    Migrated to preprocessing.metadata.length

Remaining experiments
---------------------

• Pepper noise diagnostics
• Evaluation plotting
• Other exploratory analyses

Design rule
-----------

Experimental scripts may compare, visualize, or prototype new ideas.

Canonical preprocessing must live under
src/cnt_project/preprocessing/

and should provide

- reusable APIs
- validation
- stable schemas
- configurable runners
- no dependence on train/test folder layout
- outputs stored under

    data/cnt_segmentation/metadata/

Status
------

The density, wavelet noise, and length metadata generators have been fully
migrated from the experiments package into the preprocessing package. The
remaining experiment scripts are retained only for exploratory analyses and
should not be considered canonical dataset preparation utilities.