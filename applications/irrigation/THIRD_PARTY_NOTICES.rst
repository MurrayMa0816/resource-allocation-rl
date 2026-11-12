External data and scientific references
==============================================================================

NASA POWER
--------------------

The meteorological data were obtained from the NASA POWER Project, funded
through the NASA Earth Science Division Applied Sciences Program. The frozen
weather is a regional proxy at 30.866666666666667 N, 112.18333333333334 E,
not 93 local stations. Retrieval metadata: data/hongmiao93/weather_manifest.json.
ET0, conditional forecasts and donor pools are derived project calculations.
The prior uses 2018--2021; 2022 and 2024 are historical evaluation forcing.

Dataset attribution and use information: https://registry.opendata.aws/nasa-power/
and https://creativecommons.org/licenses/by/4.0/ . Retain attribution and
identify derived calculations. This software is independent of NASA and does
not imply endorsement.

Published agricultural service areas
------------------------------------------------------------------------

regions_93.csv transcribes numerical outlet identifiers and midseason rice
areas from CN119514977A Table 2 (92 sluice outlets and one endpoint):
https://patents.google.com/patent/CN119514977A/zh . These are not recovered
GIS polygons, certified cadastral boundaries or reconstructed dispatch.
Table 3 gives 2 mm/day for the third main canal and groups seepage parameters
by main canal; it does not calibrate the 93 outlet units separately. The
U(1,3) mm/day percolation distribution and frozen assignments are project
research assumptions. No full patent PDF or source article is redistributed
in this package.

The schematic upstream canal hierarchy is derived from CN118521431B,
paragraphs [0176]--[0182], PDF pages 17--18:
https://patents.google.com/patent/CN118521431B/zh . One Hongmiao branch canal
serves the published outlet units; four drawing rows are page layout, not
four real branches. Gate symbols and service-area rectangles are original
schematic drawings without surveyed channel geometry or service boundaries.
This canal background adds no routing, conveyance or gate-control dynamics
to the field-inlet allocation environment and makes no exclusive-source claim.

Crop and soil references
------------------------------------------------

Calendar, Kc/Kcb and four-stage Jensen response: Chen et al., River (2024),
https://doi.org/10.1002/rvr2.96 . Soil reference:
https://doi.org/10.11975/j.issn.1002-6819.2018.19.015 . These are same-district
transfers, not calibration of all Hongmiao fields. Attributed numerical
parameters and research interpretations are bundled; full publications are not.
See case_model.json, source_registry.csv and manifest.json for provenance.

Initial water and research supply assumptions
------------------------------------------------------------------------------------------

The Hubei Agricultural Development Center published the 2023 guidance from
the Ministry of Agriculture and Rural Affairs Agricultural Mechanization
Station recommending 1--3 cm shallow water during establishment/recovery:
https://hbnysy.hubei.gov.cn/bmdt/ztzl/nykjcs/njnykjcs/njfw_nj/202304/t20230404_4612915.shtml .
This supports a water-depth range, not a measured distribution across Hongmiao
units. Initial h0~U(10,30) mm, saturated soil, seed 20261013 and fixed outlet
assignments are research assumptions. The patent's May 22 initial-state record
is not treated as a measured May 30 state.

The default 30,000 m3 daily quota and 2,430,000 m3 seasonal budget are project
supply controls. They use about 73% of the approximately 3,329,440.84 m3 model
reference demand for the first 81 days, May 30 through August 18, as a scale
and then adopt a rounded daily quota. They are not
historical supply measurements or evidence of feasible daily conveyance.
