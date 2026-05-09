import json
import sys

def convert(ui_path, api_path):
    with open(ui_path, 'r') as f:
        data = json.load(f)
    
    api_format = {}
    nodes = {n['id']: n for n in data['nodes']}
    links = {l[0]: l for l in data['links']} if 'links' in data else {}
    
    for n in data['nodes']:
        nid = str(n['id'])
        api_format[nid] = {
            "inputs": {},
            "class_type": n['type'],
            "_meta": {
                "title": n.get('properties', {}).get('Node name for S&R', n['type'])
            }
        }
        
        # Add widgets
        if 'widgets_values' in n:
            widgets = n['widgets_values']
            if isinstance(widgets, list):
                # We need to map list to input names.
                # In UI format, inputs are explicit if linked, but widgets are positional.
                # Since we don't have widget names easily for all nodes, we have to look at n['inputs']? No, n['inputs'] are only for links.
                # Actually, ComfyUI UI JSON saves widget values in a list. The backend API format requires dict keys.
                pass
            elif isinstance(widgets, dict):
                for k, v in widgets.items():
                    api_format[nid]['inputs'][k] = v
                    
        # This is risky because widget names are not in the UI JSON if it's a list.
        # Let's check if the UI JSON has widget names in inputs?
