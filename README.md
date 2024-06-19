<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://privkit.fc.up.pt/_static/logo_white_word.svg">
  <img alt="Privkit Logo" src="https://privkit.fc.up.pt/_static/logo_grey_word.svg">
</picture>

-----------------

# Privkit: A Privacy Toolkit

Privkit is a privacy toolkit that provides methods for privacy analysis. It includes different data types, privacy-preserving mechanisms, attacks, and metrics. The current version is focused on location data and facial data. The Python Package is designed in a modular manner and can be easily extended to include new mechanisms. Privkit can be used to process data, configure privacy-preserving mechanisms, apply attacks, and also evaluate the privacy/utility trade-off through suitable metrics.

See [https://privkit.fc.up.pt](https://privkit.fc.up.pt) for a complete documentation.

## Citation

If you use **privkit** in a scientific publication, please consider to cite:

```
@inproceedings{cunha2024privkit,
  title={Privkit: A Toolkit of Privacy-Preserving Mechanisms for Heterogeneous Data Types},
  author={Cunha, Mariana and Duarte, Guilherme and Andrade, Ricardo and Mendes, Ricardo and Vilela, Jo{\~a}o P},
  booktitle={Proceedings of the Fourteenth ACM Conference on Data and Application Security and Privacy},
  pages={319--324},
  year={2024}
}
```

## Installation

Privkit can be installed through this Github repository or by using pip:

```
pip install privkit
```

Then, if needed, you can run the following command to install the dependencies.

```
pip install -r requirements.txt
```

## Examples

The repository [https://github.com/privkit/privkit-tutorials](https://github.com/privkit/privkit-tutorials) contains practical tutorials of Privkit available as Jupyter Notebooks. This repository aims at promoting the reproducibility of results from research papers.

#### Getting started

```py
import privkit as pk

data_to_load = [['2008-10-23 02:53:04', 39.984702, 116.318417],
                ['2008-10-23 02:53:10', 39.984683, 116.31845],
                ['2008-10-23 02:53:15', 39.984686, 116.318417]]

location_data = pk.LocationData()
location_data.load_data(data_to_load, datetime=0, latitude=1, longitude=2)

planar_laplace = pk.ppms.PlanarLaplace(epsilon=0.01)
planar_laplace.execute(location_data)
```

## License

Privkit is open source and licensed under the BSD 3-clause license.