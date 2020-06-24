
.PHONY: clean

all: paper.pdf appendix.pdf

paper.pdf: optimized.pdf
	qpdf optimized.pdf --pages . 1-13 -- paper.pdf

appendix.pdf: optimized.pdf
	qpdf optimized.pdf --pages . 14-z -- appendix.pdf

optimized.pdf: output.pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=optimized.pdf output.pdf

output.pdf:
	pdflatex -jobname=output paper.tex
	bibtex output.aux
	pdflatex -jobname=output paper.tex
	pdflatex -jobname=output paper.tex

clean:
	rm *.blg *.bbl *.aux *.log

build: output.pdf
