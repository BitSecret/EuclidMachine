from em.formalgeo.tools import parse_gdl,save_readable_parsed_gdl,load_json
if __name__ == '__main__':
    save_readable_parsed_gdl(parse_gdl(load_json('mygdl.json')),
                             'parsed_gdl.json')
