ICRA 2027 manuscript notes
==========================

- root.tex is the authoritative manuscript. The other .tex entry points
  include it for compatibility and contain no independent paper text.
- The paper uses the included ieeeconf.cls with US Letter paper, 10 pt type,
  two columns, and \overrideIEEEmargins.
- The review manuscript is double-anonymous; author names and affiliations are
  omitted.
- ICRA 2027 permits at most eight pages including references. Check the final
  page count and run the PaperCept PDF compliance test before submission.
- references.bib contains an anonymized placeholder entry for SCRIBE. Replace
  it with the publication metadata appropriate at submission time.

Build from this directory with:

    latexmk -pdf root.tex
