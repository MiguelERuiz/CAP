
compile:
	latexmk -pdf

clean:
	rm images/*.dot.png
	rm -rf _minted-presentacion
	latexmk -c

dot:
	dot -T png -O images/*.dot