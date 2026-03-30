from neo4j import GraphDatabase

# Config
URI = 'bolt://localhost:7687'
USERNAME = 'neo4j'
PASSWORD = 'ax668817'
COS_SIM_THRESHOLD = 0.8


def get_all_genres(driver):
    """
    Fetch every genre from the graph
    Args:
        driver: Neo4j driver
    Returns:
        list of genre name strings
    """
    query = "MATCH (g:Genre) RETURN g.name AS name ORDER BY g.name"
    with driver.session() as session:
        result = session.run(query)
        genres = [r['name'] for r in result]

    print(f"Genres: {genres}")
    return genres

def calculate_magnitude(driver):
    query = '''
    MATCH (t:Track)
    WITH t,
        sqrt(
            t.danceability^2 + t.energy^2 + t.acousticness^2 +
            t.valence^2 + t.speechiness^2 + t.instrumentalness^2 +
            t.liveness^2 + (t.tempo / 250.0)^2 +
            ((t.loudness + 60) / 60.0)^2
        ) AS magnitude
    SET t.vector_mag = magnitude;
    '''
    with driver.session() as session:
        session.run(query)

def create_similarity_edges(driver, genre, threshold):
    """
    Compute cosine similarity for all track pairs in a genre and 
    create SIMILAR_TO edges for pairs above the threshold.
    Args:
        driver: Active Neo4j driver instance
        genre: Genre name string to process
        threshold: minimum cosine similarity score to create an edge
    Returns:
        None
    """
    query = """
        MATCH (t1:Track)-[:BELONGS_TO]->(g:Genre {name: $genre})<-[:BELONGS_TO]-(t2:Track)
        WHERE t1.track_id < t2.track_id 
        AND t1.vector_mag IS NOT NULL 
        AND t2.vector_mag IS NOT NULL
        WITH t1, t2,
            (t1.danceability * t2.danceability + 
            t1.energy * t2.energy + 
            t1.acousticness * t2.acousticness + 
            t1.valence * t2.valence + 
            t1.speechiness * t2.speechiness + 
            t1.instrumentalness * t2.instrumentalness + 
            t1.liveness * t2.liveness + 
            (t1.tempo / 250.0) * (t2.tempo / 250.0) + 
            ((t1.loudness + 60) / 60.0) * ((t2.loudness + 60) / 60.0)
            ) AS dot_product
        WITH t1, t2, dot_product / (t1.vector_mag * t2.vector_mag) AS cos_sim
        WHERE cos_sim > $threshold
        MERGE (t1)-[r:SIMILAR_TO {score: cos_sim}]->(t2)
        RETURN count(r) AS created;
    """
    with driver.session() as session:
        result = session.run(query, genre=genre, threshold=threshold)
        count = result.single()['created']

        print(count)
        return count


def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    calculate_magnitude(driver)
    genres = get_all_genres(driver)
    for g in genres:
        create_similarity_edges(driver, g, COS_SIM_THRESHOLD)
    
if __name__ == "__main__":
    main()
