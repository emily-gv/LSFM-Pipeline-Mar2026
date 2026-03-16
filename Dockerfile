FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy your environment file
COPY lsfm_pipeline.yml .

# Create the conda environment
RUN conda env create -f lsfm_pipeline.yml

# Make the environment active by default
SHELL ["conda", "run", "-n", "lsfm-pipeline", "/bin/bash", "-c"]

# Optional: set it as the default env when you open a shell
RUN echo "conda activate lsfm-pipeline" >> ~/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "lsfm-pipeline"]
CMD ["/bin/bash"]