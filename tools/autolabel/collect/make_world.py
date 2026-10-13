"""Copy a world, adding a Label plugin to every artefact so a bounding-box camera reports it.

Labels are YOLO class index + 1 (label 0 is background). Only used to collect training data.
usage: make_world.py <in.sdf> <out.sdf>
"""
import re, sys
CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
MESH = {'white_sphere': 'white_sphere', 'mushrooms_blue': 'blue_mushrooms', 'green_alien/': 'green_alien',
        'ice_castles': 'ice_formation', 'green_crystals': 'green_crystals'}
src = open(sys.argv[1]).read()
counts = {}
def label(m):
    body = m.group(0)
    cls = next((c for key, c in MESH.items() if key in body), None)
    if cls is None:
        return body
    counts[cls] = counts.get(cls, 0) + 1
    plugin = ('      <plugin filename="ignition-gazebo-label-system" '
              'name="ignition::gazebo::systems::Label">'
              f'<label>{CLASSES.index(cls) + 1}</label></plugin>')
    cut = body.rfind('</model>')
    return body[:cut].rstrip() + '\n' + plugin + '\n    </model>'
out = re.sub(r'<model name="artifact_\d+">.*?</model>', label, src, flags=re.S)
open(sys.argv[2], 'w').write(out)
print(counts)
