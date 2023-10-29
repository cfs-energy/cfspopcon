set -ex

git clone https://github.com/cfs-energy/radas.git

pushd radas

git checkout d9e23824f2edc46ef35e2fd54cf26438a3180733

poetry install --only main

poetry run python adas_data/fetch_adas_data.py

poetry run python run_radas.py all

popd
