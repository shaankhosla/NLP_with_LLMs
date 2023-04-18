import torch
from train import T5Finetuner
from transformers import AutoTokenizer
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


class TextGenerate:
    def __init__(self, model_name, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache/")
        convert_zero_checkpoint_to_fp32_state_dict(model_path, "lightning_model")
        self.training_model = T5Finetuner.load_from_checkpoint("lightning_model")
        self.model = self.training_model.model

    def generate_summary(
        self, ctext, summ_len=150, beam_search=2, repetition_penalty=2.5
    ):
        source_id, source_mask, _, _ = self.model.encode_text(self.tokenizer, ctext, "")
        self.model.eval()
        with torch.no_grad():
            # generated_ids = self.model.generate(
            #     input_ids=source_id,
            #     attention_mask=source_mask,
            #     max_length=summ_len,
            #     num_beams=beam_search,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=1.0,
            #     early_stopping=True,
            # )
            generated_ids = self.model.generate(source_id)
            prediction = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
        return prediction


if __name__ == "__main__":
    model = TextGenerate(
        "t5-small",
        "./output/logs/t5-small/version_79/checkpoints/epoch=0-step=1.ckpt",
    )
    output = model.generate_summary(
        """Engineering at Tackle.io, tackle.io/careers Computer Software Founder MadSkills.io Co-founder, CTO Twine Helped to build a very senior collaborative team that quickly solved complex problems for a Slack bot + IoT + Alexa skill product. MEAN (MongoDB, Express.js, Angular.js, Node.js) stack to build Slack bots, Alexa skills, and linux based IoT device. Director of Development MobileDay Took on more responsibility for the engineering team as the company wound down. Still heavily focused on iOS and backend development. Led/championed the project that led to a 5X increase in revenue and put the app on auto pilot. Senior Software Engineer/Founder Behind The Set LLC Behind the Set was a internet set-top box company (like Roku). I led development of the backend using Grails and XMPP as well the client framework to connect to the server. Research Assistant University of Colorado Developed high energy physics analysis code in C++, Perl and Java. Technical Advisor Chat Mode Architect Advisor McKesson One of five architects that supported a large organization across four development centers. Responsible for maintaining interoperability and consistent architecture across the 20+ applications (web, java, and mobile) while implementing continually changing government regulations, mentoring tech leads, and contributing directly to the code bases. Consultant McKesson Full stack web application development using Spring/Hibernate and GWT. Implemented a reusable architecture and framework for quickly bootstrapping applications on that stack. Engineering Tackle.io Research And Development FreeConferenceCall.com Principal Software Engineer MobileDay Upgraded the app to iOS 8 and 9, implemented extensions, watch app, and a complete redesign and conversion to swift while consistently maintaining 5 star app rating.
Developed and supported web/api functionality using MEAN stack and Django. Software Engineer BOCS * Engineering lead for VOD “back-end” using JAX-WS, Hibernate, Lucene, Spring et al
* Video transcoding using ffmpeg, mplayer and x264, automated using Perl and Groovy Software Engineer Sun B-Tier J2EE Web Service development as part of a distributed team
XML Document based Web Services using JAX-RPC/JAXB
Deploying and packaging web services for  Sun Java Application Server
Designed and implemented JUnit Tests for web services Software Engineer Echostar *Lead engineer for java services using JAX-WS, Hibernate, Lucene, Spring et al on Linux:
-Video on demand metadata/content services (IPTV)
-Web app/services for controlling remote devices
*Quick prototyping/development of SOAP web services using open source tools
*Designed and implemented servlets to deliver both binary and XML data
*Scalable/high availability web services, 1000+ connections/second
*Ported C/C++ code to java for handling proprietary binary data formats
*Scalable database design using Hibernate and PostgreSQL Techstars Alexa Accelerator - Offered in partnership with Amazon’s Alexa Fund, this program is designed to support early-stage companies advancing the state-of-the-art in voice-powered technologies, interfaces and applications, with a focus on Alexa domains such as connected home, wearables and hearables, enterprise, communication devices, connected car and wellness. University of Colorado Boulder BS, Physics Engineering, Minor: Computer Science, Minor: Math International Linear Collider group.
* Designed, developed and tested physics simulation software
- Detector resolution and analysis feasibility studies
- Developed pattern recognition software using Hessian matrices
- Analyzed multi-Terabyte data sets
- System administration and software maintenance
* Undergraduate Research Opportunity Program Grant
-Developed and tested new data analysis method and software
* Earn-Learn Program Award
- Detector simulation software design and development

Computer graphics
- Investigated projecting 4d objects into 3d spaces using OpenGL
Physics classroom demonstration
- Engineered  electro-magnetic physics demonstration
10 hours per epoch
        """
    )
    print(output)
