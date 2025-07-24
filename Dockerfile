# InSilicoVA Docker Environment
# This Dockerfile provides a reproducible environment for running InSilicoVA
# with the exact same base image used in development and testing.

# Use Ubuntu 22.04 LTS for reproducibility
FROM ubuntu:22.04

# Labels
LABEL maintainer="your.email@example.com"
LABEL description="Docker container for development of InSilicoVA-sim (R, Java, and required packages)"

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and prepare for R installation from CRAN
RUN apt-get update && apt-get install -y \
    software-properties-common \
    dirmngr \
    gnupg \
    wget \
    ca-certificates \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
    && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
    && apt-get update

# Install R from CRAN repository and other dependencies
RUN apt-get install -y \
    r-base \
    r-base-dev \
    openjdk-11-jdk \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libgit2-dev \
    pandoc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables for ARM64 architecture
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV LD_LIBRARY_PATH=/usr/lib/jvm/java-11-openjdk-arm64/lib/server

# Configure R to use Java
RUN R CMD javareconf

# Install required R packages for InSilicoVA
# First install rJava which requires Java to be configured
RUN R -e "install.packages('rJava', repos='https://cran.rstudio.com/')"

# Install additional dependencies for openVA
RUN R -e "install.packages(c('MCMCpack', 'xtable'), repos='https://cran.rstudio.com/')"

# Install InterVA packages (used in openVA pipeline)
RUN R -e "install.packages(c('InterVA4', 'InterVA5'), repos='https://cran.rstudio.com/')"

# Finally install openVA and InSilicoVA
RUN R -e "install.packages(c('openVA', 'InSilicoVA'), repos='https://cran.rstudio.com/')"

# Create working directory and volume
RUN mkdir -p /data
WORKDIR /data
VOLUME ["/data"]

# Set the default command to bash
CMD ["bash"]