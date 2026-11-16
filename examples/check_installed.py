"""Check the installed tests and full CLI outputs outside the checkout.

Use --section tests or --section commands to run either half on its own.
The default runs both. These are same-runtime packaging comparisons, not
independent numerical or statistical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['clusterlab', 'clusterlab.__main__', 'clusterlab.io',
           'clusterlab.kmeans', 'clusterlab.preprocess', 'clusterlab.metrics',
           'clusterlab.data', 'clusterlab.experiment', 'clusterlab.stability',
           'clusterlab.ambiguity']
COMMANDS = [
    ['demo'], ['train', '--out', 'refit'], ['evaluate', '--out', 'refit'],
    ['predict', 'predict-input.json'],
    ['predict', 'predict-input.json', '--models', 'refit/models.json'],
    ['stability', '--out', 'stability.json'],
    ['stability', '--models', 'refit/models.json', '--out', 'fresh-stability.json'],
    ['ambiguity', '--out', 'ambiguity.json'],
    ['ambiguity', '--stability', 'fresh-stability.json', '--out', 'fresh-ambiguity.json'],
]
FILES = ['refit/protocol.json', 'refit/models.json', 'refit/evaluation-plan.json',
         'refit/report.json', 'stability.json', 'fresh-stability.json',
         'ambiguity.json', 'fresh-ambiguity.json']


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels), str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'assets', 'docs']:
        shutil.copytree(ROOT / name, outside / name)
    # The figure test reads a frozen SVG beside the installed package.
    # Copy that static directory only, never any library Python source.
    package = Path(call([py, '-I', '-c', 'import clusterlab;print(clusterlab.__path__[0])'], work).strip())
    shutil.copytree(ROOT / 'assets', package.parent / 'assets')
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
if sys.argv[3]=='tests':
    tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun>0 and result.wasSuccessful()
    print(json.dumps(dict(tests=result.testsRun)))
check()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''
    if args.section in ['all', 'tests']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'tests'], outside))
    if args.section in ['all', 'commands']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        reference = work / 'reference'
        reference.mkdir()
        fixtures = {'predict-input.json': json.loads((ROOT / 'examples/predict.json').read_text())}
        for folder in [reference, outside]:
            for name, value in fixtures.items():
                (folder / name).write_text(json.dumps(value))
        source_env = ENV.copy()
        source_env['PYTHONPATH'] = str(ROOT)
        expected = []
        for command in COMMANDS:
            output = call([sys.executable, '-m', 'clusterlab'] + command, reference, source_env).strip()
            expected.append(output)
        saved = {name: json.loads((reference / name).read_text()) for name in FILES}
        for entry in [[py, '-I', '-m', 'clusterlab'], [str(work / 'env/bin/clusterlab')]]:
            for command, report in zip(COMMANDS, expected):
                output = call(entry + command, outside).strip()
                assert output == report, (entry, command)
            for name, report in saved.items():
                assert json.loads((outside / name).read_text()) == report, (entry, name)
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        print(json.dumps(dict(command_invocations=len(COMMANDS), command_kinds=6,
                              complete_output_files=len(FILES), fresh_models=40, entrypoints=2)))
