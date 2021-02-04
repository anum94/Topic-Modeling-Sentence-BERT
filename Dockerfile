FROM continuumio/miniconda3

#create working directory
WORKDIR /merckproject

# copy all files from current working directory to the image container
COPY . /merckproject

# listing all the contents to make sure all files have been copied successfully
RUN ls

#update conda
RUN conda update -n base -c defaults conda

# create enviroment using enviroment.yml
RUN conda env create -f enviroment.yml
#RUN conda create -n env python=3.6

RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/nlp/bin:$PATH

#Make port 80 available to the world outside this container
ENV PORT 5000


#define the enviroment variable
ENV NAME MerckWorld

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]

#docker run --rm -it -p 5000:5000/tcp testimage:tag6
