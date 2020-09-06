import flask
import difflib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = flask.Flask(__name__, template_folder = 'templates')


mov_df = pd.read_csv('./data/preprocessed_tmdb.csv')

tag_df = mov_df.drop(columns = ['original_title', 'overview', 'movie_id'])
tag_df['TAGS'] = tag_df[tag_df.columns[0:3]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
tag_df.drop(columns = ['genres', 'keywords', 'cast'], inplace = True)

count = CountVectorizer(stop_words='english')
c_matrix = count.fit_transform(tag_df['TAGS'])
tfidf = TfidfVectorizer(stop_words='english')
t_matrix = tfidf.fit_transform(mov_df['overview'].apply(lambda x: np.str_(x)))
s_matrix = sp.hstack([c_matrix, t_matrix], format='csr')
cos_sim = cosine_similarity(s_matrix, s_matrix)

All_Mov_titles = [ mov_df['original_title'][i] for i in range(len(mov_df['original_title']))]
All_Mov_titles = [x.lower() for x in All_Mov_titles]
indc = pd.Series(mov_df.index , index=All_Mov_titles)


def recommendation(name):
    index = indc[name]
    similarity_score = list(enumerate(cos_sim[index]))
    similarity_score = sorted(similarity_score, key = lambda x: x[1], reverse = True)
    similarity_score = similarity_score[1: 21]
    #print(similarity_score)
    mov_index = [i[0] for i in similarity_score]
    #print(mov_index)
    tit = mov_df['original_title'].iloc[mov_index]
    genr = mov_df['genres'].iloc[mov_index]
    #print(tit)
    web = mov_df['homepage'].iloc[mov_index]
    rating = mov_df['vote_average'].iloc[mov_index]
    date = mov_df['release_date'].iloc[mov_index]
    final_rec = pd.DataFrame(columns=['Title', 'Rating', 'Genres', 'Release', 'WebLinks'])
    final_rec['Title'] = tit
    final_rec['Rating'] = rating
    final_rec['Genres'] = genr
    final_rec['Release'] = date
    final_rec['WebLinks'] = web
    return final_rec

#print(recommendation('avatar'))

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        chk_name = m_name.lower()
        m_name = m_name.title()
#        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if chk_name not in All_Mov_titles:
            return(flask.render_template('error.html',name=m_name))
        else:
            result_final = recommendation(chk_name)
            names = []
            rate = []
            genres = []
            link = []
            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                rate.append(result_final.iloc[i][1])
                genres.append(result_final.iloc[i][2])
                dates.append(result_final.iloc[i][3])
                link.append(result_final.iloc[i][4])

            return flask.render_template('positive.html',movie_names=names,movie_rate=rate,movie_genr=genres,movie_date=dates,movie_link=link, search_name=m_name)

if __name__ == '__main__':
    app.run()
