import setuptools as st
from pip._internal.req import parse_requirements

install_reqs = parse_requirements("requirements.txt", session="reqs")
reqs = [str(ir.requirement) for ir in install_reqs]
st.setup(name='twittermatchmaker',
      version='1.0',
      description='extract twitter dataset from list of claims',
      author='Michael Shliselberg',
      author_email='michael.shliselberg@uconn.edu',
      url='https://github.com/RIET-lab/TwitterMatchMaker',
      package_dir={"": "src"},
      packages=st.find_packages("src"),
      install_requires=install_reqs
)