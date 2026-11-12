Figure 6.7 native compact layout
================================

Run ``python plot_compact.py`` beside the two existing published budget CSVs.
Use ``--output-dir DIR`` for isolated reproduction. The script uses the local
thesis FigureText font (or explicit ``--font PATH``), draws native 10.5 bp
text, and reads only the existing summary values. It does not load a model,
execute an environment, train, infer, or create evaluation data.

Canvas is 105 x approximately 79.55 mm; plotting axes are 85 x 65 mm. Values,
axes limits, ticks, colors, marker and error-bar styles, and all three current
annotation data anchors and display offsets are unchanged from layout.pdf.
The accepted 350 label offset remains (0,28) bp; the 400/450 labels remain
(-20,12)/(-23,12) bp. There is no new annotation exception.

Outputs are compact.pdf/svg/png, compact_geometry.json, and
compact_numeric_artists.json. Existing original figure/layout files, CSVs and
source are retained. Installing this variant requires only the new seven-file
copy list in the task's audit/figure67_compact_freeze.json; no old asset is
replaced. A parent-controlled TeX change may then select compact.pdf.
