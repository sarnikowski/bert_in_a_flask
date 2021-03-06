FROM tensorflow/tensorflow:latest-gpu-py3

RUN useradd -ms /bin/bash palpatine

SHELL ["/bin/bash", "-c"]

RUN USER=palpatine && \
    GROUP=palpatine && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt --no-cache-dir

USER palpatine:palpatine

ENTRYPOINT ["fixuid"]
