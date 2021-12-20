FROM armindocachada/tensorflow2-opencv4-raspberrypi4:2.2_4.5.0
RUN pip3 install pandas

# Set the working directory
WORKDIR /work

# Add shell-script
ENTRYPOINT ["/work/run.sh" , "/work"]
