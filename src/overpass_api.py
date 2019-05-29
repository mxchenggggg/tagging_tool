import overpy

class Overpass_api:
	def __init__(self, b_box):
		self.b_box = b_box
		self.api = overpy.Overpass()
		self.result = self.api.query("""
		way({},{},{},{});
		(._;>;);
    	out body;
		""".format(self.b_box[0], self.b_box[1], self.b_box[2], self.b_box[3]))
		for way in self.result.ways:
			print("name: {}".format(way.tags.get("name", "n/a")))