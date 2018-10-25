TARGET=DenseBoxV1_alan

all: 
	pdflatex $(TARGET).tex
	#detex $(TARGET).tex | grep -v '^$$' > $(TARGET).txt
	bibtex $(TARGET)
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex
	rm -f $(TARGET).synctex $(TARGET).aux $(TARGET).log $(TARGET).bbl $(TARGET).blg $(TARGET).brf $(TARGET).txt $(TARGET).out

clean:
	rm -f $(TARGET).synctex $(TARGET).aux $(TARGET).pdf $(TARGET).log $(TARGET).bbl $(TARGET).blg $(TARGET).brf $(TARGET).txt $(TARGET).out
	rm -f *.log
	find . -name '*.eps' -exec rm -v {} \; 

show: all
	evince $(TARGET).pdf
