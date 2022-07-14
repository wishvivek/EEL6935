# Project Overview

This project compares the REST API architectural style against the SOAP design.

* **Target Audience**: Software engineers, developers with prior knowledge or experience with web services and API design.
* **Project Scope**: This project mainly Highlights when you should consider the SOAP or REST architectural style, with only a brief overview of their fundamentals. For more details, please check the official [SOAP](https://www.w3.org/TR/soap/) documentation and the [REST API Wikipedia](https://en.wikipedia.org/wiki/Representational_state_transfer).

# Overview of SOAP

* The Simple Object Access Protocol is a standard communications protocol that encodes data in XML to define an extensive, standardised, and secure messaging framework that facilitates structured information to be exchanged in a decentralised, distributed environment.
* A SOAP message contains 3 components: Envelope, Header, and Body
* SOAP API calls are stateful, i.e., the server stores information about the client and uses that information over a series of requests or chain of operations.
* SOAP is independent of an underlying transport protocol, so there are no restrictions on using HTTP or some other transport protocol.
* SOAP prevents unauthorised access to the messages (user data) received by and sent from SOAP APIs based on the Web Standards (WS) Security principles and specifications that include mechanisms such as passwords, XML encryption, and security tokens, among others.
* SOAP supports several WS specifications, such as WS-Security, WS-ReliableMessaging, WS-Coordination, WS-AtomicTransaction, to name a few. See the official W3C documentation for more information.
* SOAP provides support for identity verification through intermediaries rather than just point-to-point security.
* SOAP offers built-in retry logic to compensate for failed communications, i.e., if there’s a problem with your request, the response contains error information that you can use to fix the problem.


## Overview of REST

* Representational State Transfer is a simpler and more flexible architectural style of building APIs, compared to SOAP, and can transfer data in a variety of formats, XML, HTML, JSON, and plain text.
* An API is **RESTful** when it constitutes the following design facets:
    * A Uniform Interface (resource, URI, and HTTP information)
    * Statelessness: all client-server communications are stateless
    * “Cacheability”: caching at client side
    * Layering: layers can exist between client and server
* REST APIs return all relevant information about the requested resource (in its current state) in a standardised format that clients can easily interpret.
* Because REST API calls are stateless, every request-response is independent and provides all the information required to complete that interaction.
* REST only supports traditional web security mechanisms like HTTPS. So, when an application sends and retrieves a message from a REST API using HTTPS, the message is secured only for the HTTPS connection only during the transport.  
* Unlike SOAP, REST APIs don’t have the built-in WS security capabilities or extensions and their security depends on the design of the APIs themselves.


## Deciding Between SOAP and REST

So, which one is the best for you? It depends on your application, business needs, and many more factors, since they both have their pros & cons. However, here are some suggestions to help you make that decision:

### When to use SOAP

* **Asynchronous, private APIs**: SOAP allows data to be transferred in a decentralised, distributed environment, ensures message-level security, and supports strong web security mechanisms, making it especially ideal for enterprise solutions	
* **Stateful operations**: Despite the higher requirements of server resources and bandwidth, the statefulness of SOAP becomes crucial when performing repetitive or chained tasks, like bank transfers.
* **Flexible transport protocol**: SOAP allows easy customization of the transport protocol as per your business need, e.g., using SMTP (Simple Mail Transfer Protocol) or JMS (Java Messaging Service), or some other protocol.
* **Sensitive applications**: Since SOAP supports several WS specifications, it is ideal for internal data transfers and other security-sensitive tasks.
* **Programming language used**: In some programming languages, you need to build SOAP XML requests manually, which becomes problematic because SOAP is intolerant of errors. However, languages like .NET reduce the effort required to create the request and to parse the response. So, this trade-off is addressed largely by the language you choose. 
* **Built-In Error Handling**: SOAP offers built-in error handling, so no “error guessing” is needed. Moreover, error reporting even provides standardized codes, allowing the possibility of automating some error handling tasks in your code.
* **Standardized design**: Because SOAP is regulated by the W3C organization, it allows increased accuracy, easy collaboration, and formal contracts.


### When to use REST

* **Developing public APIs**: Compared to SOAP, REST APIs are considered easier to use and adopt, which makes them ideal for creating public web services, wherein security requirements are lower
* **Fast, efficient design**: REST APIs can use different data formats, like JSON, XML, or HTML, making them faster and more efficient than most SOAP APIs
* **Limited resources & bandwidth**: Since REST API calls are stateless, the server does not store information on past requests, which reduces the server memory needed and improves performance due to reduced overhead 
* **Flexible authentication design**: Although REST does not have the built-in WS security capabilities, it supports common, highly customisable authentication methods such as HTTP basic authentication, JSON web tokens, OAuth, and API keys. This is in contrast to SOAP that is bound by the WS Security specifications. 
* **Building mobile applications**: Because REST is lightweight, efficient, stateless, and cacheable, it is ideal for building mobile applications. 
* **Fast model design**: REST allows easier integration with existing websites, with no need for refactoring site infrastructure. This enables you to work faster, with an easier learning curve, rather than spend time rewriting a site from scratch. Instead, you can simply add additional functionality.
* **“Malicious inputs” protections**:  Because REST rejects any request that violates user specifications, you can customise configurations that protect the underlying web application from malformed and malicious inputs, even after the client has gained access.


## Further Reading

* https://stackify.com/soap-vs-rest/
* https://www.infoq.com/articles/rest-soap-when-to-use-each/
* https://smartbear.com/blog/soap-vs-rest-whats-the-difference/
* https://blog.dreamfactory.com/when-to-use-rest-vs-soap-with-examples/

