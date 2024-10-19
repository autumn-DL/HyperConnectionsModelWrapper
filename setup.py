from setuptools import setup, find_packages

setup(
    name='hyper-connections-wrapper',  # 包的名称（通常使用小写字母和连字符）
    version='0.1.0',                   # 版本号
    author='autumn_dl',                # 作者名称
    author_email='your.email@example.com',  # 作者邮箱
    description='A brief description of HyperConnectionsWrapper',  # 简要描述
    long_description=open('README.md').read(),  # 从 README 文件读取详细描述
    long_description_content_type='text/markdown',  # 描述格式
    url='https://github.com/yourusername/hyper-connections-wrapper',  # 项目的 URL
    packages=find_packages(),  # 自动发现并列出包
    classifiers=[              # 分类器
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 支持的 Python 版本
    install_requires=[         # 依赖包列表
        'torch',               # 例如，PyTorch
    ],
)