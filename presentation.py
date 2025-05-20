from test_case_gen import gen_requirements_pdf_to_test_case
import base64

path_to_pdf = "ReqView-Example_Software_Requirements_Specification_SRS_Document.pdf"
with open(path_to_pdf, "rb") as f:
    pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

output = gen_requirements_pdf_to_test_case(pdf_data)
print(output)