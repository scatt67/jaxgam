# Citations

jaxgam is a Python port of Simon Wood's
[mgcv](https://cran.r-project.org/package=mgcv) R package. The
statistical methods are entirely his work. If you use jaxgam, please
cite the relevant mgcv papers.

## References

**GAM method (REML/ML estimation)**

> Wood SN (2011). "Fast stable restricted maximum likelihood and marginal
> likelihood estimation of semiparametric generalized linear models."
> *Journal of the Royal Statistical Society (B)*, 73(1), 3--36.
> [doi:10.1111/j.1467-9868.2010.00749.x](https://doi.org/10.1111/j.1467-9868.2010.00749.x)

**Beyond exponential family**

> Wood SN, Pya N, Säfken B (2016). "Smoothing parameter and model
> selection for general smooth models (with discussion)." *Journal of the
> American Statistical Association*, 111, 1548--1575.
> [doi:10.1080/01621459.2016.1180986](https://doi.org/10.1080/01621459.2016.1180986)

**GCV-based model method and basics of GAMM**

> Wood SN (2004). "Stable and efficient multiple smoothing parameter
> estimation for generalized additive models." *Journal of the American
> Statistical Association*, 99(467), 673--686.
> [doi:10.1198/016214504000000980](https://doi.org/10.1198/016214504000000980)

**Overview**

> Wood SN (2017). *Generalized Additive Models: An Introduction with R*
> (2nd ed.). Chapman and Hall/CRC.

**Thin plate regression splines**

> Wood SN (2003). "Thin-plate regression splines." *Journal of the Royal
> Statistical Society (B)*, 65(1), 95--114.
> [doi:10.1111/1467-9868.00374](https://doi.org/10.1111/1467-9868.00374)

## BibTeX

```bibtex
@Article{wood2011,
  title = {Fast stable restricted maximum likelihood and marginal
    likelihood estimation of semiparametric generalized linear models},
  journal = {Journal of the Royal Statistical Society (B)},
  volume = {73},
  number = {1},
  pages = {3--36},
  year = {2011},
  author = {S. N. Wood},
  doi = {10.1111/j.1467-9868.2010.00749.x},
}

@Article{wood2016,
  title = {Smoothing parameter and model selection for general smooth
    models (with discussion)},
  author = {S. N. Wood and N. Pya and B. S{\"a}fken},
  journal = {Journal of the American Statistical Association},
  year = {2016},
  pages = {1548--1575},
  volume = {111},
  doi = {10.1080/01621459.2016.1180986},
}

@Article{wood2004,
  title = {Stable and efficient multiple smoothing parameter estimation
    for generalized additive models},
  journal = {Journal of the American Statistical Association},
  volume = {99},
  number = {467},
  pages = {673--686},
  year = {2004},
  author = {S. N. Wood},
  doi = {10.1198/016214504000000980},
}

@Book{wood2017,
  title = {Generalized {A}dditive {M}odels: An Introduction with {R}},
  year = {2017},
  author = {S. N. Wood},
  edition = {2},
  publisher = {Chapman and Hall/CRC},
}

@Article{wood2003,
  title = {Thin-plate regression splines},
  journal = {Journal of the Royal Statistical Society (B)},
  volume = {65},
  number = {1},
  pages = {95--114},
  year = {2003},
  author = {S. N. Wood},
  doi = {10.1111/1467-9868.00374},
}
```

## Machine-readable citation

See [CITATION.cff](https://github.com/scatt67/jaxgam/blob/main/CITATION.cff)
for machine-readable citation metadata. GitHub renders a "Cite this
repository" button from this file.

## License

jaxgam is licensed under the
[GPL-2.0-or-later](https://github.com/scatt67/jaxgam/blob/main/LICENSE),
matching mgcv's `GPL (>= 2)` license. As a derivative work of mgcv, this
ensures downstream users have the same freedoms granted by the original
package.
