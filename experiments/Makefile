
manifests.tar.gz:
	$(eval TEMP := $(shell mktemp))
	# find experiments -type f -name "*.json" ! -name 'config.json' | sort > "${TEMP}"

	# latest mnist-like experiments
	find "mnist-like" -type f -name 'config.json' \
		| sort | grep "grids-20200213" \
		> "${TEMP}"

	# legacy mnist-like experiments
	find "mnist-like" -type f -name 'config.json' \
		| sort | grep "legacy__mnist-like" \
		>> "${TEMP}"

	# latest CIFAR10 experiments
	find "cifar/grids/grids-short" -type f -name 'config.json' \
		| sort \
		>> "${TEMP}"

	# k-3 experiments
	find "musicnet/grids/grid-slow/" -type f -name 'config.json' \
		| sort \
		>> "${TEMP}"

	# full experiments
	find "musicnet/grids/grid-fast/" -type f -name 'config.json' \
		| sort \
		>> "${TEMP}"

	# experiments with fine-tune
	find "musicnet/grids/musicnet__threshold__fine-tune/" -type f -name 'config.json' \
		| sort \
		>> "${TEMP}"

	# tar from list, dropping `/config`
	tar -czf manifests.tar.gz --transform s/\\/config// -T "${TEMP}"
	rm "${TEMP}"
