vers = ['std', 'adv']
n_layer = 12

langs = ["fr", "de", "es", "fa", "it", "ru", "pt", "zh-CN", "tr", "ar", "et", "mn", "nl", "sv-SE", "lv", "sl", "ta", "ja", "id"]

for ver in vers:
    for layer_id in range(n_layer):
        md_str = "|  |"

        for lang in langs:
            md_str = md_str + " {} |".format(lang)

        md_str = md_str + "\n|" + "----|" * (len(langs) + 1)

        for i, lang_i in enumerate(langs):
            md_str = md_str + "\n|{}|".format(lang_i)
            for j, lang_j in enumerate(langs):
                if i < j:
                    md_str = md_str + "<embed src=\"{}/layer_{}/{}_{}_{}.pdf\" width=\"500\" height=\"375\" type=\"application/pdf\">|".format(
                        ver, layer_id, ver, lang_i, lang_j)
                elif i > j:
                    md_str = md_str + "<embed src=\"{}/layer_{}/{}_{}_{}.pdf\" width=\"500\" height=\"375\" type=\"application/pdf\">|".format(
                        ver, layer_id, ver, lang_j, lang_i)
                else:
                    md_str = md_str + '|'
        
        with open('md/{}_layer_{}.md'.format(ver, layer_id), 'w') as w:
            w.write(md_str)