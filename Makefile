
.PHONY: arxiv-submission.zip

arxiv-submission.zip:
	# prepare a temporary directory
	$(eval target := $(shell mktemp -d))

	# copy assets and clean up
	cp -r ./assets "${target}"/assets
	find "${target}"/assets -type f ! -name "*.pdf" -delete

	# copy the paper and references
	cp ./paper/paper.tex ./paper/references.bib "${target}"/

	# patch in new path
	printf "17c\n\\graphicspath{{./assets/}}\n.\n" | patch -e "${target}"/paper.tex

	# build pdf
	cd "${target}" ; \
		pdflatex paper.tex ; \
		bibtex paper.aux ; \
		pdflatex paper.tex ; \
		pdflatex paper.tex

	# clean up
	cd "${target}" ; \
		rm paper.pdf ; \
		rm *.blg *.aux *.log

	# zip it and cleanup
	$(eval zip := $(shell mktemp))
	cd "${target}"; \
		zip -9qr "${zip}.zip" . -x ".*" -x "__MACOSX"

	mv "${zip}.zip" arxiv-submission.zip

	rm -rf "${target}"
