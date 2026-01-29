FROM dizcza/docker-hashcat:latest

WORKDIR /app

# Copy only the hashcat-related runner
COPY run_hashcat_benchmark.sh /app/run_hashcat_benchmark.sh
RUN chmod +x /app/run_hashcat_benchmark.sh

ENTRYPOINT ["/app/run_hashcat_benchmark.sh"]