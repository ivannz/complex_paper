supplement="supp"

.PHONY: paper experiments code

submission: paper $(supplement).zip
	if [ -d "submission" ]; then rm -rf "submission"; fi
	mkdir submission

	mv "$(supplement).zip" ./submission

	cp paper/paper.pdf paper/appendix.pdf ./submission

$(supplement).zip: code paper experiments
	# prepare a temporary directory
	$(eval target := $(shell mktemp -d))

	# copy generated figures and notebooks for the experiments
	cp -r ./experiments "${target}"/
	# find "${target}" -type f ! -iname "*.ipynb" -delete
	cp -r ./experiments/manifests.tar.gz "${target}/experiments"

	# put a sdist of the package into it
	cp -p "`ls -dtr1 ./dist/*.tar.gz | tail -1`" "${target}"/

	# copy generated figures and notebooks for the experiments
	cp -r ./assets "${target}"/

	# copy generated figures and notebooks for the experiments
	cp -r ./README.md "${target}"/

	# ship appendix with the supplement
	cp paper/appendix.pdf "${target}"/

	# zip it and cleanup
	$(eval zip := $(shell mktemp))
	cd "${target}"; \
		zip -9qr "${zip}.zip" . -x ".*" -x "__MACOSX"
	mv "${zip}.zip" "$(supplement).zip"

	rm -rf "${target}"

paper:
	make -C paper all

code:
	python setup.py sdist

experiments:
	make -C experiments
