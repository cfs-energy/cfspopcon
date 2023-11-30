set -ex

git clone https://github.com/cfs-energy/radas.git

pushd radas

git checkout b538ea3d42f0d3eea6cf28433a2390503457083d

poetry install --only main

poetry run python adas_data/fetch_adas_data.py

poetry run python run_radas.py all

popd
