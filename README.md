<h1 align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/smlab-niser/tirtha-public/main/media/images/tirtha-logo-dark.webp" media="(prefers-color-scheme: dark)">
        <img src="https://raw.githubusercontent.com/smlab-niser/tirtha-public/main/media/images/tirtha-logo-light.webp" width=25 height=25>
        </picture>
    $\color{#ff4c40}{\textrm{Project Tirtha [Beta]}}$
    <!-- Project Tirtha [Beta] -->
</h1>

<div align="center">
    <a href="https://www.niser.ac.in" target="_blank">
        <picture>
            <source srcset="./media/images/niser-logo-dark.webp" media="(prefers-color-scheme: dark)">
            <img src="./media/images/niser-logo-light.webp" width=150 height=150>
        </picture>
    </a>
    <a href="https://www.lafondation3ds.org/" target="_blank">
        <picture>
            <source srcset="./media/images/lfds-logo-dark.webp" media="(prefers-color-scheme: dark)">
            <img src="./media/images/lfds-logo-light.webp" width=200 height=150>
        </picture>
    </a>
    <a href="https://odisha.gov.in/explore-odisha/state-archaeology" target="_blank">
        <picture>
            <source srcset="./media/images/odisha-logo-dark.webp" media="(prefers-color-scheme: dark)">
            <img src="./media/images/odisha-logo-light.webp" width=130 height=150>
        </picture>
    </a>
</div>

---

[![Signal Group](https://img.shields.io/badge/Signal-%23039BE5.svg?&style=for-the-badge&logo=Signal&logoColor=white)](https://signal.group/#CjQKIN_Ry9rBYUZJn8pLEkfWMAkZvO2FGopPalXsWPMZauXyEhBT1GdJYb5C_PJV0qE5VTLj) [![Element Chat Room](https://img.shields.io/matrix/tirtha%3Amatrix.org?style=for-the-badge&logo=element)](https://matrix.to/#/#tirtha:matrix.org) [![GitHub Discussions](https://img.shields.io/github/discussions/smlab-niser/tirtha-public?style=for-the-badge&logo=github)](https://github.com/smlab-niser/tirtha-public/discussions)

---

> [!important]
> Please go here for the new Tirtha site: https://smlab.niser.ac.in/project/tirtha/.

---

> [!important]
> **Google Summer of Code aspirants**, please go here for the project topics: https://docs.google.com/document/d/1p5UxgoKBhy5pQh3fXX00BftcmJxzxdFLOkrB3Hibfm4.

---
[Project Tirtha](https://smlab.niser.ac.in/project/tirtha/) is an *academic initiative* to create
3D models of heritage sites using crowdsourced images. The word *Tirtha* is Sanskrit
for "a place of pilgrimage", and is commonly used to refer to the sacred sites of
Hinduism, Jainism and Buddhism. Our goal is to preserve and showcase the beauty and cultural 
significance of heritage sites. We believe that by allowing the general public to contribute to
the creation of these models, and by providing open access to these models, we can
increase awareness and appreciation of these important cultural landmarks and inspire
future generations to maintain them for years to come.

This project is now open-source under the [GNU Affero General Public License v3.0](./LICENSE)
and is under active development. All contributions are welcome. Please read
[CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

See [Citation](#citation) for information on how to cite this project. A 
[CITATION.cff](./CITATION.cff) file is also available in the repository.

## System Architecture

<picture>
    <source srcset="./media/images/architecture-dark.webp" media="(prefers-color-scheme: dark)">
    <img src="./media/images/architecture-light.webp" alt="Tirtha Broad Architecture">
</picture>

[See the paper](#citation) for more details.

## Requirements
### Hardware
* **OS**: Ubuntu 22.04 LTS (Other Linux distros may work, but are not tested)
* **RAM**: 16 GB+ for modestly sized image sets (< 500 images)
* **VRAM**: 8 GB+; NVIDIA GPU required for CUDA support
* **CPU**: 16+ cores recommended
* **Storage**: 100 GB+ free space recommended

### Software
* **Primary**: `Python 3.11` (Developed using `Python 3.11.7`) & `Node.js` >= 18.0.0 & `npm` >= 8.0.0.
* The build process (manual or docker) automatically installs these and other dependencies. Check [Deployment / Development Setup](#deployment--development-setup) below for more details.

## Deployment / Development Setup

> [!tip]
> Please go [here](https://github.com/smlab-niser/tirtha-docker) to set up Tirtha using Docker.
> We **strongly recommend** using Docker for deployment as well as for testing or development.

### Manual Setup
* Clone the repository and `cd` to the `tirtha-public` directory:
    ```sh
    git clone https://github.com/smlab-niser/tirtha-public.git
    cd tirtha-public
    ```
* Edit `tirtha-public/build/tirtha.env` to set the environment variables.
* **Carefully go through** `build.sh` and edit to ensure that the paths and settings are correct for your system. For instance, you can choose to not install some of the dependencies for the backend, which saves a considerable amount of time.
* The default build uses ports 8000 (for gunicorn), 8001 (for Postgres), and 15672 (for RabbitMQ) on the host system. Ensure these ports are free. If you want to use different ports, you will have to edit `tirtha.env`, `gunicorn.conf.manual.py`, and `tirtha.docker.nginx` before running `build.sh`.
* `cd` to `tirtha-public/build` and run the following command to set up Tirtha. This will install the required packages and set up Postgres, RabbitMQ, Nginx, Gunicorn and Tirtha:
    ```bash
    cd tirtha-public/build
    sudo bash build.sh
    ```
* To run Tirtha, use the following command:
    ```bash
    bash start.sh
    ```
* The Tirtha web interface can be accessed at `http://localhost:8000` or `http://<HOST_IP>:8000` if you are setting up Tirtha on a remote server. To access the Django admin interface, use `http://localhost:8000/admin` or `http://<HOST_IP>:8000/admin`. The default username and password can be found in the `tirtha.env` file.
* To access Tirtha-related logs, check the `/var/www/tirtha/logs/` directory. Logs for system packages, like RabbitMQ or Postgres, can be accessed using `journalctl`.
* If you want to set up SSL for your Tirtha instance, check the [tirtha.ssl.nginx](https://github.com/smlab-niser/tirtha-public/blob/main/tirtha_bk/config/tirtha.ssl.nginx) configuration file.
* To set up system service and socket for Tirtha, you can refer to the [tirthad.docker.service](https://github.com/smlab-niser/tirtha-public/blob/main/tirtha_bk/config/tirthad.docker.service) and [tirthad.docker.socket](https://github.com/smlab-niser/tirtha-public/blob/main/tirtha_bk/config/tirthad.docker.socket) files.

> [!important]
> 1. Currently, the production directory is hard-coded to `/var/www/tirtha`.
> Changing this will require changes to the [`nginx` configuration](https://github.com/smlab-niser/tirtha-public/blob/main/tirtha_bk/config/tirtha.docker.nginx) and the [`gunicorn.conf.manual.py`](https://github.com/smlab-niser/tirtha-public/blob/main/tirtha_bk/gunicorn/gunicorn.conf.manual.py) file, along with `build.sh`.
> 2. You may also have to configure your firewall to allow traffic on the ports used by Tirtha. Check the `tirtha.env` file and the nginx configuration for the ports used.

## Citation
Please cite the following paper if you use this software in your work ([arXiv](https://arxiv.org/abs/2308.01246) | [Papers with Code](https://paperswithcode.com/paper/tirtha-an-automated-platform-to-crowdsource) | [ACM Digital Library](https://dl.acm.org/doi/10.1145/3611314.3615904)):
```bibtex
@inproceedings{10.1145/3611314.3615904,
    author = {Shivottam, Jyotirmaya and Mishra, Subhankar},
    title = {Tirtha - An Automated Platform to Crowdsource Images and Create 3D Models of Heritage Sites},
    year = {2023},
    isbn = {9798400703249},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3611314.3615904},
    doi = {10.1145/3611314.3615904},
    abstract = {Digital preservation of Cultural Heritage (CH) sites is crucial to protect them against damage from natural disasters or human activities. Creating 3D models of CH sites has become a popular method of digital preservation thanks to advancements in computer vision and photogrammetry. However, the process is time-consuming, expensive, and typically requires specialized equipment and expertise, posing challenges in resource-limited developing countries. Additionally, the lack of an open repository for 3D models hinders research and public engagement with their heritage. To address these issues, we propose Tirtha, a web platform for crowdsourcing images of CH sites and creating their 3D models. Tirtha utilizes state-of-the-art Structure from Motion (SfM) and Multi-View Stereo (MVS) techniques. It is modular, extensible and cost-effective, allowing for the incorporation of new techniques as photogrammetry advances. Tirtha is accessible through a web interface at https://smlab.niser.ac.in/project/tirtha/ and can be deployed on-premise or in a cloud environment. In our case studies, we demonstrate the pipeline’s effectiveness by creating 3D models of temples in Odisha, India, using crowdsourced images. These models are available for viewing, interaction, and download on the Tirtha website. Our work aims to provide a dataset of crowdsourced images and 3D reconstructions for research in computer vision, heritage conservation, and related domains. Overall, Tirtha is a step towards democratizing digital preservation, primarily in resource-limited developing countries.},
    booktitle = {Proceedings of the 28th International ACM Conference on 3D Web Technology},
    articleno = {11},
    numpages = {15},
    keywords = {photogrammetry, open source, digital heritage, crowdsourcing, 3D dataset},
    location = {San Sebastian, Spain},
    series = {Web3D '23}
}
```
You can also use GitHub's citation feature to generate a citation for this repository. See [here](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for more details.

## Funding & Acknowledgment
This project is funded by [La Fondation Dassault Systèmes](https://www.lafondation3ds.org/). We also thank the following individuals for their contributions to the project's development:
- [JeS24](https://github.com/JeS24)
- [AvTheBlackBird](https://github.com/AvTheBlackBird)
- [annadapb](https://github.com/annadapb)

We are grateful to the developers of the following open-source libraries, which help make this project a reality:
- [AliceVision Meshroom](https://github.com/alicevision/Meshroom/), available under the [Mozilla Public License 2.0](https://github.com/alicevision/Meshroom/blob/develop/LICENSE-MPL2.md).
- [MANIQA](https://github.com/IIGROUP/MANIQA), available under the [Apache 2.0 License](https://github.com/IIGROUP/MANIQA/blob/master/LICENSE).
- [nsfw_model](https://github.com/GantMan/nsfw_model), available under the [MIT License](https://github.com/GantMan/nsfw_model/blob/master/LICENSE.md).
- [obj2gltf](https://github.com/CesiumGS/obj2gltf)
- [gltfpack](https://github.com/zeux/meshoptimizer)
- [model-viewer](https://github.com/google/model-viewer)
- [Google Fonts | Rubik](https://github.com/googlefonts/rubik)
- [Google Fonts | Material Icons](https://github.com/google/material-design-icons)
- [Django](https://github.com/django/django)
- [jQuery](https://github.com/jquery/jquery)
- [gunicorn](https://github.com/benoitc/gunicorn)
- [nginx](https://github.com/nginx/nginx)
- [Celery](https://github.com/celery/celery)
- [RabbitMQ](https://github.com/rabbitmq)
- [OpenCV](https://github.com/opencv/opencv)
- [Docker](https://github.com/docker)

We also thank Odisha State Archaeology for their support.

---

&copy; 2023-24 Project Tirtha,
[Subhankar Mishra's Lab](https://www.niser.ac.in/~smishra/),
[School of Computer Sciences](https://oldsite.niser.ac.in/scps/), NISER.
All rights reserved.
