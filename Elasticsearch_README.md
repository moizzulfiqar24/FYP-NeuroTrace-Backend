# Elasticsearch Docker Setup

Set up and run **Elasticsearch** locally using Docker for development and testing purposes.


## Prerequisites

- [Docker](https://www.docker.com/) must be installed and running on your machine.

## 🚀 Getting Started

### 🔧 Run Elasticsearch in Docker

Use the following command to pull and run Elasticsearch 8.13.4 in a Docker container:

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.4
```

Port: Exposes Elasticsearch on localhost:9200
Security: Disabled for ease of local use
Heap Size: 2 GB min and max heap

### Verifying the Setup

Wait 20–30 seconds after running the container, then check the status using:

```bash
curl http://localhost:9200
```

ℹ️ If you get an error like Recv failure: Connection reset by peer, give it a few seconds and retry.

Once ready, you’ll receive a response like:

```bash
{
  "name": "fee37053ebc0",
  "cluster_name": "docker-cluster",
  "cluster_uuid": "7WmiKxoKTD2xWM8hxq-O4w",
  "version": {
    "number": "8.13.4",
    "build_flavor": "default",
    "build_type": "docker",
    "build_hash": "da95df118650b55a500dcc181889ac35c6d8da7c",
    "build_date": "2024-05-06T22:04:45.107454559Z",
    "build_snapshot": false,
    "lucene_version": "9.10.0",
    "minimum_wire_compatibility_version": "7.17.0",
    "minimum_index_compatibility_version": "7.0.0"
  },
  "tagline": "You Know, for Search"
}
```

## Stopping the Container

To stop the Elasticsearch container:

```bash
docker stop elasticsearch
```

## Restarting Elasticsearch

To start it again:

```bash
docker start elasticsearch
```

Then verify with:

```bash
curl http://localhost:9200
```