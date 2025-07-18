# Dockerfile for InSilicoVA Table 3 Replication
# Build with: docker build -f Dockerfile.insilicova -t insilicova-arm64:latest .

FROM rocker/r-ver:4.4.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('openVA', 'dplyr', 'devtools'), dependencies=TRUE, repos='https://cloud.r-project.org/')"

# Verify installations
RUN R -e "library(openVA); library(dplyr); cat('All packages loaded successfully\n')"

# Set working directory
WORKDIR /data

# Default command
CMD ["R"]