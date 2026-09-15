;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "root"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("ieeeconf" "letterpaper" "10pt" "conference")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "") ("amsfonts" "") ("amssymb" "") ("graphicx" "") ("cite" "") ("balance" "") ("algorithm" "") ("algpseudocode" "") ("hyperref" "bookmarks=true" "hidelinks")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "ieeeconf"
    "ieeeconf10"
    "amsmath"
    "amsfonts"
    "amssymb"
    "graphicx"
    "cite"
    "balance"
    "algorithm"
    "algpseudocode"
    "hyperref")
   (LaTeX-add-labels
    "fig:front-reconstructions"
    "eq:target-measure"
    "eq:inference-problem"
    "alg:arrodes"
    "eq:som-worlds"
    "eq:eof-field"
    "eq:occupation"
    "eq:mmd"
    "eq:mmd-expanded"
    "eq:energy"
    "eq:maturity"
    "eq:som-posterior"
    "eq:som-prior"
    "eq:eof-posterior"
    "eq:proposal-covariance"
    "eq:transport"
    "eq:transport-weight"
    "eq:som-distance"
    "fig:world-space"
    "fig:final-recovery"
    "fig:detailed-recovery"
    "fig:recovery-histories")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

