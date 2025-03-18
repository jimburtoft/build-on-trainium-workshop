# Build On Trainium Workshop

In this workshop you will learn how to develop support for a new model with [NeuronX Distributed Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-overview.html#nxdi-overview), through the context of Llama 3.2 1B. You will also learn how to write your own kernel to directly program the accelerated hardware with the [Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html). Both of these tools will help you design your research proposals and experiments on Trainium.

### What is Build on Trainium? 
Build on Trainium is a $110M credit program focused on AI research and university education to support the next generation of innovation and development on AWS Trainium. AWS Trainium chips are purpose-built for high-performance deep learning (DL) training of generative AI models, including large language models (LLMs) and latent diffusion models. Build on Trainium provides compute credits to novel AI research on Trainium, investing in leading academic teams to build innovations in critical areas including new model architectures, ML libraries, optimizations, large-scale distributed systems, and more. This multi-year initiative lays the foundation for the future of AI by inspiring the academic community to utilize, invest in, and contribute to the open-source community around Trainium. Combining these benefits with Neuron software development kit (SDK) and recent launch of the Neuron Kernel Interface (NKI), AI researchers can innovate at scale in the cloud.

### What are AWS Trainium and Neuron?
AWS Trainium is an AI chip developed by AWS for accelerating building and deploying machine learning models. Built on a specialized architecture designed for deep learning, Trainium accelerates the training and inference of complex models with high output and scalability, making it ideal for academic researchers looking to optimize performance and costs. This architecture also emphasizes sustainability through energy-efficient design, reducing environmental impact. Amazon has established a dedicated Trainium research cluster featuring up to 40,000 Trainium chips, accessible via Amazon EC2 Trn1 instances. These instances are connected through a non-blocking, petabit-scale network using Amazon EC2 UltraClusters, enabling seamless high-performance ML training. The Trn1 instance family is optimized to deliver substantial compute power for cutting-edge AI research and development. This unique offering not only enhances the efficiency and affordability of model training but also presents academic researchers with opportunities to publish new papers on underrepresented compute architectures, thus advancing the field.

Learn more about Build On Trainium [here](https://aws.amazon.com/ai/machine-learning/trainium/research/).

### Your workshop
This hands-on workshop is designed for academic researchers who are planning on submitting proposals to [Build On Trainium](https://www.amazon.science/research-awards/call-for-proposals). 

The workshop has 3 main modules:
1. Set up instructions
2. Run inference with Llama and NeuronX Distributed inference (NxD)
3. Write your own kernel with Neuron Kernel Interface (NKI)

#### Instructor-led workshop
If you are participating in an instructor-led workshop, follow the guidance provided by your instructor for accessing the environment.

#### Self-managed workshop
If you are following the workshop steps in your own environment, you will need to take the following actions:
1. Launch a trn1.2xlarge instance on Amazon EC2, using the latest DLAMI with Neuron packages preinstalled
2. Use a Python virtual environment preinstalled in that DLAMI, commonly located in `/opt/aws_<xxx>`.
3. Set up and manage your own development environment on that instance, such as by using VSCode or a Jupyter Lab server.

### Background knowledge
This workshop introduces developing on AWS Trainium for the academic AI research audience. As such it's expected that the audience will already have a firm understanding of machine learning fundamentals. 

### Workshop costs
If you are participating in an instructor-led workshop hosted in an AWS-managed Workshop Studio environment, you will not incur any costs through using this environment. If you following this workshop in your own environment, then you will incur associated costs with provisioning an Amazon EC2 instance. Please see the service pricing details [here](https://aws.amazon.com/ec2/pricing/on-demand/). 

At the time of writing, this workshop uses a trn1.2xlarge instance with an on-demand hourly rate in supported US regions of $1.34 per hour.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

