# InSilicoVA Docker Environment
# This Dockerfile provides a reproducible environment for running InSilicoVA
# with the exact same base image used in development and testing.

# Use the specific image SHA256 for reproducibility
FROM ubuntu@sha256:61df64731dec9b9e188b2b5a78000dd81e63caf78957257872b9ac1ba947efa4

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install R and dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libgit2-dev \
    pandoc \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages for InSilicoVA
RUN R -e "install.packages(c('openVA', 'InSilicoVA'), repos='https://cran.rstudio.com/')"

# Create working directory
WORKDIR /data

# Set the default command to R
CMD ["R"]