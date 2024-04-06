from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

# 更新课程亮点提示模板以包括课程概要
course_highlight_template = """本课程，{course_name}，将带您从3D建模的基础概念走向高级技巧，专注于'挖洞'与'卡线'方法，让您的创作生动起来。课程概要：{course_overview}"""
course_highlight_prompt = PromptTemplate.from_template(course_highlight_template)

# Introduction to the instructor and the course
instructor_intro_template = """Meet your instructor, a seasoned 3D artist and expert in Maya, here to guide you through the wonders of 3D modeling."""
instructor_intro_prompt = PromptTemplate.from_template(instructor_intro_template)

# Motivational closing
closing_statement_template = """Embark on this learning adventure to unlock your creative potential and transform your digital artistry. Let's create something incredible together!"""
closing_statement_prompt = PromptTemplate.from_template(closing_statement_template)


# Define the full template that incorporates all parts
full_course_intro_template = """{instructor_intro}

{course_highlight}

{closing_statement}"""
full_course_intro_prompt = PromptTemplate.from_template(full_course_intro_template)

# 使用更新后的 course_highlight_prompt，其余部分保持不变
input_prompts = [
    ("instructor_intro", instructor_intro_prompt),
    ("course_highlight", course_highlight_prompt),
    ("closing_statement", closing_statement_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_course_intro_prompt, pipeline_prompts=input_prompts
)

# Define the full template that incorporates all parts
full_course_intro_template = """{instructor_intro}

{course_highlight}

{closing_statement}"""
full_course_intro_prompt = PromptTemplate.from_template(full_course_intro_template)

# 使用更新后的 course_highlight_prompt，其余部分保持不变
input_prompts = [
    ("instructor_intro", instructor_intro_prompt),
    ("course_highlight", course_highlight_prompt),
    ("closing_statement", closing_statement_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_course_intro_prompt, pipeline_prompts=input_prompts
)