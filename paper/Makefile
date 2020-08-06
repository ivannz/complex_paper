acrobat = "/Applications/Adobe Acrobat DC/Adobe Acrobat.app"

.PHONY: clean

all: paper.pdf appendix.pdf

paper.pdf: optimized.pdf
	qpdf optimized.pdf --pages . 1-13 -- paper.pdf

appendix.pdf: optimized.pdf
	qpdf optimized.pdf --pages . 14-z -- appendix.pdf

optimized.pdf: output.pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=optimized.pdf output.pdf

	# open in acrobat pro and wait until it is closed
	if [ -d ${acrobat} ]; \
		then open -Wa ${acrobat} optimized.pdf; \
	fi

output.pdf:
	pdflatex -jobname=output paper.tex
	bibtex output.aux
	pdflatex -jobname=output paper.tex
	pdflatex -jobname=output paper.tex

clean:
	rm *.blg *.bbl *.aux *.log

build: output.pdf
