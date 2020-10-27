HOCKING-chapter.pdf: HOCKING-chapter.tex refs.bib
	rm -f *.aux *.log *.bbl
	pdflatex HOCKING-chapter
	bibtex HOCKING-chapter
	pdflatex HOCKING-chapter
	pdflatex HOCKING-chapter
