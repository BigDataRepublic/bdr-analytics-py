# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = bdr-analytics-py
SOURCEDIR     = .
BUILDDIR      = _build

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

clean:
	rm -Rf source
	rm -Rf $(BUILDDIR)

# This target depends on all .py files in bdranalytics folder
# Thus if any one changed, also the api doc will be updated
# If there are py files for which no apidoc is created, using -f does not work:
# None of the apidocs will be updated (nothing changed), but the py file will
# be newer (as there is no corresponding api doc file)
# Therefore we remove the source dir first
#source: ../bdranalytics
source:	$(shell find ../bdranalytics -type f -name '*.py')
	-rm -Rf source
	sphinx-apidoc -f -M -T -o source/ -H "$(SPHINXPROJ)" ../bdranalytics

html: Makefile source
	mkdir -p $(BUILDDIR)/html $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."
