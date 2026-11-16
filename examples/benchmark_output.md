Mean over 8 synthetic leaves (192x192) per damage type.

| damage | method | PSNR in hole (dB) | SSIM whole image | leaf shape IoU in hole | time (s) |
| --- | --- | --- | --- | --- | --- |
| bites | no restoration | 6.0 | 0.967 | 0.300 | 0.00 |
| bites | harmonic | 13.8 | 0.983 | 0.603 | 0.00 |
| bites | biharmonic | 13.7 | 0.984 | 0.663 | 0.01 |
| bites | telea | 12.0 | 0.980 | 0.674 | 0.00 |
| bites | navier_stokes | 11.9 | 0.980 | 0.715 | 0.00 |
| bites | exemplar | 15.7 | 0.988 | 0.892 | 0.52 |
| holes | no restoration | 5.1 | 0.968 | 0.522 | 0.00 |
| holes | harmonic | 29.2 | 0.995 | 1.000 | 0.00 |
| holes | biharmonic | 28.1 | 0.994 | 1.000 | 0.01 |
| holes | telea | 28.2 | 0.993 | 1.000 | 0.00 |
| holes | navier_stokes | 29.0 | 0.994 | 1.000 | 0.00 |
| holes | exemplar | 33.9 | 0.999 | 0.999 | 0.34 |
| tear | no restoration | 5.3 | 0.971 | 0.357 | 0.00 |
| tear | harmonic | 22.7 | 0.992 | 0.949 | 0.00 |
| tear | biharmonic | 23.4 | 0.992 | 0.962 | 0.01 |
| tear | telea | 21.2 | 0.990 | 0.947 | 0.00 |
| tear | navier_stokes | 21.4 | 0.990 | 0.957 | 0.00 |
| tear | exemplar | 24.8 | 0.996 | 0.981 | 0.40 |
