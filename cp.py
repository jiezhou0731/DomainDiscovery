# -*- coding: UTF-8 -*-
#!/usr/bin/python
from bs4 import BeautifulSoup
import re
import json
import requests
from requests.auth import HTTPBasicAuth
from urlparse import urlparse
from urlparse import urljoin
import urllib
import os.path
import socket
import urllib2
import re

socket.setdefaulttimeout(5)
class CP :
	total = 5715697
	sampleRate = 1

	def __init__(self):
		self.sampleTotal = self.total * self.sampleRate
		self.size = 1000
		self.sampleTime = self.sampleTotal / self.size

	def downloadWeaponDataFromCDR(self):
		for i in range(0,int(self.sampleTime)):
			print 1.0 * i/self.sampleTime
			data = {"from" : i/self.sampleTime*total, "size" : size,"query" : {"match_all" : {}}};
			data_json = json.dumps(data)
			headers = {'content-type': 'application/json'}
			r = requests.post('https://els.istresearch.com:19200/memex-domains/weapons/_search', data=data_json, auth=HTTPBasicAuth('memex', 'qRJfu2uPkMLmH9cp'), headers=headers)
			f = open("dump/"+str(i)+'.json','w')
			f.write(r.text.encode('utf-8'))
			f.close()

	def getAllCDRVisitedUrls(self):
		domains = {}
		for i in range(0,int(self.sampleTime)):
			print str(1.0 * i/self.sampleTime) + '\t' + str(len(domains))
			with open("dump/"+str(i)+".json") as json_file:
				json_data = json.load(json_file)
				docCount=0
				

				# For each document in the json file
				for doc in json_data["hits"]["hits"]:
					html = unicode(doc["_source"]["raw_content"])
					d={"cdr_data":""}
					d['cdr_data'] = json.dumps(doc["_source"])
					d['_index'] = str(doc["_index"])
					d['_type'] = str(doc["_type"])
					d['id'] = str(doc["_id"])
					if (doc["_source"]["url"] is list):
						d['url'] = doc["_source"]["url"][0]
					else:
						d['url'] = doc["_source"]["url"]

					d['url'] = ''.join(d['url'])
					d['url'] = d['url'].partition('?')[0]
					url = d['url'] 
					parsed_uri = urlparse( url )
					domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
					if domain in domains:
						domains[domain] = domains[domain]+1
					else:
						domains[domain] = 1
		f = open('visitedDomains.txt','w')
		f.write("total: "+str(len(domains))+'\n')
		for k,v in domains.items():
			f.write(k + ' \t' + str(v) +'\n')
		f.close()

	def getDomainFromUrl(self, url):
		parsed_uri = urlparse( url )
		domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
		return domain
	def getAllLinkingUrls(self):
		domains = {}
		sampleTime = 10
		chunkSize = self.total / self.size / sampleTime
		for i in range(0,sampleTime):
			print str(1.0 * i/sampleTime) + '\t' + str(len(domains))
			with open("dump/"+str(i*chunkSize)+".json") as json_file:
				json_data = json.load(json_file)
				docCount=0
				

				# For each document in the json file
				for doc in json_data["hits"]["hits"]:
					html = unicode(doc["_source"]["raw_content"])
					soup = BeautifulSoup(html, "lxml")
					anchors = soup.findAll('a')
					# For each anchor
					for anchor in anchors:
						try :
							if not anchor['href'].startswith('http'):
								continue
							domain = self.getDomainFromUrl(anchor['href'])
						except:
							pass
					if domain in domains:
						domains[domain] = domains[domain]+1
					else:
						domains[domain] = 1
					
		f = open('linkingDomains.txt','w')
		f.write("total: "+str(len(domains))+'\n')
		for k,v in domains.items():
			f.write(k + ' \t' + str(v) +'\n')
		f.close()
	def generateTrainData(self):
		domains={}
		with open('linkingDomains.txt') as openfileobject:
			for line in openfileobject:
				domains[line.split(" ", 2)[0].strip()] = 1
		with open('visitedDomains.txt') as openfileobject:
			for line in openfileobject:
				domains[line.split(" ", 2)[0].strip()] = 2
		with open('seedDomains.txt') as openfileobject:
			for line in openfileobject:
				domains[line.split(" ", 2)[0].strip()] = 3

		
		f = open('train.txt','w')
		for k,v in domains.items():
			print k
			if k.startswith("http"):
				currentUrl = k
			else :
				currentUrl = "http://"+k
			try:
				page=urllib2.urlopen(currentUrl)
			except:
				currentUrl = "https://"+k
				try:
					page=urllib2.urlopen(currentUrl)
				except:
					continue
			soup = BeautifulSoup(page.read(), "lxml")
			text = self.getWebsiteContent(soup)
			f.write(str(v) +', "'+k+ ' '+ text + '"'+'\n')
		f.close()

	def generateTestData(self):
		f = open('testData.txt','w')
		with open('test.txt') as openfileobject:
			for line in openfileobject:
				k=line.strip()
				print k
				if k.startswith("http"):
					currentUrl = k
				else :
					currentUrl = "http://"+k
				try:
					page=urllib2.urlopen(currentUrl)
				except:
					currentUrl = "https://"+k
					try:
						page=urllib2.urlopen(currentUrl)
					except:
						continue
				soup = BeautifulSoup(page.read(), "lxml")
				text = self.getWebsiteContent(soup)
				f.write('"'+k+'" , "'+k+ ' '+ text + '"'+'\n')
		f.close()

	def getWebsiteContent(self,soup):
		texts = soup.findAll(text=True)
		def visible(element):
			if element.parent.name in ['style', 'script','a']:
				return False
			elif re.match('<!--.*-->', str(element.encode('utf-8').strip())):
				return False
			elif re.match('\n', str(element.encode('utf-8').strip())): 
				return False
			return True
		visible_texts = filter(visible, texts)
		text = ""
		for s in visible_texts :
			if s!='\n':
				text+=s+' '
		text = text.replace('"',' ')
		text = text.replace('\n',' ')
		text = text.replace('\'',' ')
		text = text.replace(',',' ')
		text = text.encode('utf-8').strip()
		text = re.sub('[^a-zA-Z0-9\n\.]',' ',text)
		text = re.sub(' +',' ',text)
		text = text[:1000]
		return text	
